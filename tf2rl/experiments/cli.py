"""Compatible entry point; assembly lives in train.py and loops in training/."""
from tf2rl.experiments.config import imitation_parser as build_parser
from tf2rl.experiments.train import run_imitation as run
from tf2rl.experiments.train import main as _main

def main(argv=None, algorithm=None):
    return _main(argv, algorithm=algorithm, runner="imitation")

if __name__ == "__main__":
    main()
