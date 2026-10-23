"""Compatibility entry point for the sac experiment."""
from tf2rl.experiments.cli import main

if __name__ == "__main__":
    import sys
    if any(arg.split("=", 1)[0] in ("--env-backend", "--num-envs") for arg in sys.argv[1:]):
        from tf2rl.experiments.off_policy import main as vector_main
        vector_main(algorithm="sac")
    else:
        main(algorithm="sac")
