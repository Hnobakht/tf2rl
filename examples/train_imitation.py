"""Run eqn-airl through the shared experiment assembly."""

from tf2rl.experiments.train import main

if __name__ == "__main__":
    main(runner="imitation", algorithm="eqn-airl")
