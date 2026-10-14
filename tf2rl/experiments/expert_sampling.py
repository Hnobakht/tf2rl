"""Expert/noise minibatches with the historical sampling without replacement."""
import random

import numpy as np


class ExpertSampler:
    def __init__(self, data, batch_size, noise_ratio=0.):
        if not 0 <= noise_ratio < 1:
            raise ValueError("discriminator_noise must lie in [0, 1)")
        self.data = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        lengths = {len(value) for value in self.data.values()}
        if len(lengths) != 1:
            raise ValueError("Expert arrays must have equal lengths")
        self.batch_size = batch_size
        self.noise_size = int(batch_size * noise_ratio)
        self.expert_size = batch_size - self.noise_size
        if next(iter(lengths)) < self.expert_size:
            raise ValueError("Too few expert transitions for sampling without replacement")
        if not all(np.isfinite(value).all() for value in self.data.values()):
            raise ValueError("Expert data contains non-finite values")
        self.indices = range(next(iter(lengths)))

    def sample(self, replay, random_replay=None):
        if random_replay is None:
            indices = np.asarray(random.sample(self.indices, self.expert_size))
            expert = {key: value[indices] for key, value in self.data.items()}
        else:
            expert = random_replay.sample(self.expert_size)
        if self.noise_size:
            noise = replay.sample(self.noise_size)
            expert = {key: np.concatenate((expert[key], noise[key]), axis=0) for key in self.data}
        return expert
