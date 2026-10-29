"""
MIT License

Copyright (c) 2017 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Modified by keiohta

import numpy as np


class EmpiricalNormalizer:
    """Normalize mean and variance of values based on emprical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(
        self, shape, batch_axis=0, eps=1e-2, dtype=np.float32, until=None, clip_threshold=None
    ):
        """Initialize observation statistics, update limits, and optional clipping.

        Args:
            shape (tuple[int, ...]): Shape of one normalized sample, excluding the batch axis.
            batch_axis (int): Axis indexing independent samples when updating normalization statistics.
            eps (float): Positive stabilizer added to the running variance before taking its square
                root.
            dtype (numpy.dtype or type): Requested array element type.
            until (int or None): Stop learning normalization statistics after this sample count.
            clip_threshold (float or None): Optional symmetric bound applied to normalized observations.

        Returns:
            None: Initializes the instance.
        """
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self._var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        """Return a copy of the running observation mean without the batch axis.

        Returns:
            np.ndarray: Copy of the running mean, shaped like one observation without the batch axis.
        """
        return np.squeeze(self._mean, self.batch_axis).copy()

    @property
    def std(self):
        """Return the running observation standard deviation without the batch axis.

        Returns:
            np.ndarray: Running standard deviation shaped like one observation, without adding eps.
        """
        return np.sqrt(np.squeeze(self._var, self.batch_axis))

    @property
    def _std_inverse(self):
        """Return the cached inverse standard deviation with numerical stabilization.

        Returns:
            np.ndarray: Cached (variance+eps)**(-0.5), retaining the singleton batch axis for
                broadcasting.
        """
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them

        Args:
            x (array_like): Input samples; leading dimensions index samples and the remaining dimensions
                contain features.

        Returns:
            None.
        """

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = np.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = np.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        # clear cache
        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.

        Normalization coefficients are read before any requested update. Call experience first
        to initialize statistics; this method returns early when count is zero.

        Args:
            x (array_like): Input samples; leading dimensions index samples and the remaining dimensions
                contain features.
            update (bool): Update statistics after reading the current normalization coefficients;
                ignored while count is zero.

        Returns:
            np.ndarray: Normalized and optionally clipped samples with the same shape as x; the input is
                returned unchanged when no samples have been observed.
        """
        if self.count == 0:
            return x

        mean = np.broadcast_to(self._mean, x.shape)
        std_inv = np.broadcast_to(self._std_inverse, x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = np.clip(normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        """Undo observation normalization using the current mean and variance.

        Args:
            y (array_like): Normalized samples to transform back into the original observation scale.

        Returns:
            np.ndarray: Samples in the original scale, with the same shape as y. Clipped information
                cannot be recovered.
        """
        mean = np.broadcast_to(self._mean, y.shape)
        std = np.broadcast_to(np.sqrt(self._var + self.eps), y.shape)
        return y * std + mean


import numpy as np
import tensorflow as tf


class Normalizer:
    """
    Normalize input data online. This is based on following:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self, mean_only=False):
        """Initialize TensorFlow running statistics and the mean-only update mode.

        Args:
            mean_only (bool): Track the mean without updating the variance.

        Returns:
            None: Initializes the instance.
        """
        self._mean_only = mean_only
        self._n = tf.Variable(0, dtype=tf.float32)
        self._mean = tf.Variable(0, dtype=tf.float32)
        self._mean_diff = tf.Variable(0, dtype=tf.float32)
        if not self._mean_only:
            self._var = tf.Variable(0, dtype=tf.float32)

    def observe(self, x):
        """Compute next mean and std

        Args:
            x (array_like): One scalar/vector observation; observe counts this entire value as one
                sample.

        Returns:
            None.
        """
        self._n.assign_add(1)
        numerator = x - self._mean
        self._mean.assign_add((x - self._mean) / self._n)
        self._mean_diff.assign_add(numerator * (x - self._mean))
        if not self._mean_only:
            self._var = tf.clip_by_value(tf.math.divide_no_nan(self._mean_diff, self._n), 1e-2, 1e2)

    def normalize(self, x):
        """Standardize inputs using TensorFlow running statistics with safe division.

        This historical implementation requires the variance field, which is not allocated when
        mean_only=True.

        Args:
            x (array_like): One scalar/vector observation; observe counts this entire value as one
                sample.

        Returns:
            tf.Tensor: Standardized values with safe division by the tracked standard deviation.
        """
        std = tf.math.sqrt(self._var)
        return tf.math.divide_no_nan(x - self._mean, std)


class NormalizerNumpy:
    """Track online NumPy mean and variance for observation normalization."""

    def __init__(self):
        """Initialize the running sample count, mean, and variance.

        Returns:
            None: Initializes the instance.
        """
        self._n = 0
        self._mean = 0
        self._mean_diff = 0
        self._var = 0

    def observe(self, x):
        """Update the running sample count, mean, and variance with one observation.

        Args:
            x (array_like): One scalar/vector observation; observe counts this entire value as one
                sample.

        Returns:
            None.
        """
        self._n += 1
        numerator = x - self._mean
        self._mean += (x - self._mean) / self._n
        self._mean_diff += numerator * (x - self._mean)
        self._var = self._mean_diff / self._n

    def normalize(self, x, update=False):
        """Standardize observations with running statistics, optionally updating them first.

        Args:
            x (array_like): One scalar/vector observation; observe counts this entire value as one
                sample.
            update (bool): Update running statistics from the supplied observations before normalizing.

        Returns:
            np.ndarray or numpy scalar: Standardized input using the running mean and stabilized
                standard deviation.
        """
        if update:
            self.observe(x)
        return (x - self._mean) / (np.sqrt(self._var) + 1e-8)

    def get_params(self):
        """Return the running count, mean, accumulated squared deviations, and variance.

        Returns:
            tuple: Running (count, mean, accumulated_squared_deviations, variance) suitable for
                set_params.
        """
        return self._n, self._mean, self._mean_diff, self._var

    def set_params(self, n, mean, mean_diff, var):
        """Restore the running normalization statistics from saved values.

        Args:
            n (int): Number of observations represented by the restored running statistics.
            mean (array_like): Mean values or running means aligned with the input coordinates.
            mean_diff (array_like): Accumulated sum of squared deviations used by the online variance
                estimator.
            var (array_like): Variance values to restore.

        Returns:
            None.
        """
        self._n = n
        self._mean = mean
        self._mean_diff = mean_diff
        self._var = var
