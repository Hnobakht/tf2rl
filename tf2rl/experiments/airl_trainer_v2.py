"""Compatibility AIRL entry point for the shared DAC training pipeline."""
from tf2rl.experiments.irl_trainer_v2 import IRLTrainerV2


class AIRLTrainerV2(IRLTrainerV2):
    def __init__(self, policy, env, args, irl, expert_obs, expert_next_obs, expert_act,
                 expert_logp=None, test_env=None, **kwargs):
        if not irl.requires_log_probs:
            raise ValueError("AIRLTrainerV2 requires an AIRL discriminator")
        super().__init__(policy, env, args, irl, expert_obs, expert_next_obs, expert_act,
                         test_env=test_env, expert_logp=expert_logp, **kwargs)
