#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m tf2rl.experiments.train --runner off_policy --help
python -m tf2rl.experiments.train --runner on_policy --help
python -m tf2rl.experiments.train --runner imitation --help
python examples/train_sac.py --help
python examples/train_ppo.py --help
python examples/train_imitation.py --help
python examples/custom_algorithm.py --help
python -m tf2rl.experiments.train --runner imitation --algorithm sac --env-name Pendulum-v1 --max-steps 4 --n-warmup 2 --batch-size 2 --units 8,8 --memory-capacity 32 --test-interval 4 --test-episodes 1 --episode-max-steps 5 --save-test-path --logdir results/smoke-expert
