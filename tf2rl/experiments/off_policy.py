"""Compatible entry point; assembly lives in train.py and loops in training/."""
from tf2rl.experiments.config import off_policy_parser as build_parser
from tf2rl.experiments.train import run_off_policy as run
from tf2rl.experiments.train import main as _main
from tf2rl.experiments.build import make_off_policy_agent as make_policy
from tf2rl.training.checkpoints import checkpoint_prefix

def main(argv=None, algorithm=None):
    return _main(argv, algorithm=algorithm, runner="off_policy")

if __name__ == "__main__":
    main()
