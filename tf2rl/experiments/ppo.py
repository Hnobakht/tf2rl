"""Compatibility entry point for PPO through the shared on-policy launcher."""

from tf2rl.experiments.on_policy import evaluate
from tf2rl.experiments.on_policy import main as on_policy_main

__all__ = ["evaluate", "main"]


def main(argv=None):
    """Keep existing PPO commands available with their original defaults."""
    return on_policy_main(argv, algorithm="ppo")


if __name__ == "__main__":
    main()
