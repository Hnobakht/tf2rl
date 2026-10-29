"""Compatible entry point; assembly lives in train.py and loops in training/."""
from tf2rl.experiments.config import on_policy_parser as build_parser
from tf2rl.experiments.train import run_on_policy as run
from tf2rl.experiments.train import main as _main
from tf2rl.experiments.build import make_on_policy_agent as make_policy, register_on_policy_algorithm, _ALGORITHMS
from tf2rl.training.evaluation import evaluate_on_policy as evaluate

def main(argv=None, algorithm=None):
    return _main(argv, algorithm=algorithm, runner="on_policy")

if __name__ == "__main__":
    main()
