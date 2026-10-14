#!/usr/bin/env bash
set -euo pipefail
python -m tf2rl.experiments.cli --help
python examples/airl_run.py --help
python examples/gail_run.py --help
python examples/airl_eql_run.py --help
python examples/gail_eql_run.py --help
python examples/sac_run.py --env-name Pendulum-v1 --max-steps 4 --n-warmup 2 --batch-size 2 --units 8,8 --memory-capacity 32 --test-interval 4 --test-episodes 1 --episode-max-steps 5 --save-test-path --logdir results/smoke-expert
