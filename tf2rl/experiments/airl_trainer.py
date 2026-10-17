"""Compatibility AIRL entry point for the shared DAC training pipeline."""
from tf2rl.experiments.irl_trainer import IRLTrainer


class AIRLTrainer(IRLTrainer):
    """Train an AIRL reward model alongside an off-policy generator."""
    def __init__(self, policy, env, args, irl, expert_obs, expert_next_obs, expert_act,
                 expert_logp=None, test_env=None, **kwargs):
        """Configure training, replay storage, evaluation, and checkpoint handling."""
        if not irl.requires_log_probs:
            raise ValueError("AIRLTrainer requires an AIRL discriminator")
        super().__init__(policy, env, args, irl, expert_obs, expert_next_obs, expert_act,
                         test_env=test_env, expert_logp=expert_logp, **kwargs)
