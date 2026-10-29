"""Compatibility imports for vector off-policy training."""
from tf2rl.training.off_policy import OffPolicyTrainer as VectorOffPolicyTrainer
from tf2rl.training.collectors import numpy, physical_next_observations
