"""Compatibility entry point for the gail experiment."""
from tf2rl.experiments.cli import main

if __name__ == "__main__":
    main(algorithm="gail")
