"""Compatibility imports for vector on-policy training."""
from tf2rl.training.on_policy import OnPolicyTrainer as VectorOnPolicyTrainer
from tf2rl.utils.returns import rollout_targets
