"""Compatibility imports for imitation component construction."""
from tf2rl.experiments.build import (
    make_imitation_agent as make_policy, make_discriminator, make_imitation_trainer, _csv,
)
from tf2rl.training.checkpoints import restore_discriminator
