import numpy as np
import tensorflow as tf


class Normalizer:
    """
    Normalize input data online. This is based on following:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self, mean_only=False):
        """Initialize TensorFlow running statistics and the mean-only update mode."""
        self._mean_only = mean_only
        self._n = tf.Variable(0, dtype=tf.float32)
        self._mean = tf.Variable(0, dtype=tf.float32)
        self._mean_diff = tf.Variable(0, dtype=tf.float32)
        if not self._mean_only:
            self._var = tf.Variable(0, dtype=tf.float32)

    def observe(self, x):
        """Compute next mean and std

        Args:
            x: float
                Input data.
        """
        self._n.assign_add(1)
        numerator = x - self._mean
        self._mean.assign_add((x - self._mean) / self._n)
        self._mean_diff.assign_add(numerator * (x - self._mean))
        if not self._mean_only:
            self._var = tf.clip_by_value(
                tf.math.divide_no_nan(self._mean_diff, self._n), 1e-2, 1e+2)

    def normalize(self, x):
        """Standardize inputs using TensorFlow running statistics with safe division."""
        std = tf.math.sqrt(self._var)
        return tf.math.divide_no_nan(x - self._mean, std)


class NormalizerNumpy:
    """Track online NumPy mean and variance for observation normalization."""
    def __init__(self):
        """Initialize the running sample count, mean, and variance."""
        self._n = 0
        self._mean = 0
        self._mean_diff = 0
        self._var = 0

    def observe(self, x):
        """Update the running sample count, mean, and variance with one observation."""
        self._n += 1
        numerator = x - self._mean
        self._mean += (x - self._mean) / self._n
        self._mean_diff += numerator * (x - self._mean)
        self._var = self._mean_diff / self._n

    def normalize(self, x, update=False):
        """Standardize observations with running statistics, optionally updating them first."""
        if update:
            self.observe(x)
        return (x - self._mean) / (np.sqrt(self._var) + 1e-8)

    def get_params(self):
        """Return the running count, mean, accumulated squared deviations, and variance."""
        return self._n, self._mean, self._mean_diff, self._var

    def set_params(self, n, mean, mean_diff, var):
        """Restore the running normalization statistics from saved values."""
        self._n = n
        self._mean = mean
        self._mean_diff = mean_diff
        self._var = var
