"""Compatibility imports; implementation lives in ``tf2rl.algos.imitation.losses``."""
from tf2rl.algos.imitation.losses import (
    EPSILON as EPSILON,
    airl_loss as airl_loss,
    gail_loss as gail_loss,
    singular_penalty as singular_penalty,
    agent_gradient_penalty as agent_gradient_penalty,
    js_divergence as js_divergence,
)
