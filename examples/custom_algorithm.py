"""Register a PPO variant without changing collection or training loops.

python examples/custom_algorithm.py --env-name Pendulum-v1 --iterations 2
"""

from tf2rl.algos.rl.ppo import PPO
from tf2rl.experiments.build import register_on_policy_algorithm
from tf2rl.experiments.train import main


class ConservativePPO(PPO):
    """Example research variant using tighter policy-change thresholds."""

    def __init__(self, **kwargs):
        kwargs.setdefault("clip_ratio", 0.1)
        kwargs.setdefault("target_kl", 0.02)
        super().__init__(**kwargs)


if __name__ == "__main__":
    register_on_policy_algorithm("conservative_ppo", ConservativePPO, default_epochs=10)
    main(runner="on_policy", algorithm="conservative_ppo")
