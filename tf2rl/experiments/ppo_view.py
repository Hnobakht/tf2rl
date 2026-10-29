"""Compatibility entry point for the shared on-policy checkpoint viewer."""

from tf2rl.experiments.on_policy_view import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
