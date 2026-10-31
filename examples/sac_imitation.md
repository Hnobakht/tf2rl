# SAC expert data, imitation learning, and checkpoint playback

Run these commands in **Bash from the repository root**, with the project installed
and the virtual environment activated:

```bash
source .venv-modern/bin/activate
```

See [installation](../README.md#installation) if the environment is not set up.
This guide uses `Pendulum-v1` to keep the workflow small. Training budgets below
are starting examples; inspect evaluation returns before treating a policy as an
expert. For PPO experts collected with GPU simulation, use the
[PPO/MJWarp guide](ppo_mjwarp_expert.md).

## 1. Train SAC and collect complete expert episodes

```bash
python examples/train_sac.py \
  --env-name Pendulum-v1 --gpu -1 --seed 7 \
  --max-steps 100000 --n-warmup 1000 \
  --batch-size 128 --units 64 64 --memory-capacity 100000 \
  --checkpoint-interval 20000 --eval-interval 10000 --eval-episodes 5 \
  --save-expert --expert-episodes 20 --stochastic-expert \
  --expert-dir results/examples/expert_data \
  --output results/examples/sac_expert
```

SAC learns from the environment reward. After training, a separate evaluation
environment collects 20 complete episodes with the frozen policy. These are
expert demonstrations, rather than the replay buffer or random warmup transitions.
`--stochastic-expert` samples policy actions; omit it to collect deterministic
evaluation actions instead. Both modes store SAC densities evaluated at the
executed actions, which the AIRL loader requires.

The output locations are:

```text
results/examples/
├── sac_expert/
│   ├── config.json
│   ├── metrics.json           # Evaluation returns and expert-export report
│   ├── checkpoints/          # Managed policy checkpoints
│   ├── checkpoint.index      # Final checkpoint; accompanied by a .data file
│   └── events/               # TensorBoard events
└── expert_data/
    └── step_*_epi_*_return_*.pkl
```

Each `.pkl` is a compressed joblib dictionary with chronological `obs`, `act`,
`next_obs`, `rew`, `done`, `terminated`, `truncated`, and `logp` arrays, plus format
and provenance metadata. `next_obs` is the physical successor, including at an
episode boundary. Use a new expert directory for a different task or dataset.

To collect another dataset from an existing SAC checkpoint without training:

```bash
python examples/train_sac.py \
  --env-name Pendulum-v1 --gpu -1 --units 64 64 \
  --evaluate --model-dir results/examples/sac_expert \
  --eval-episodes 5 --save-expert --expert-episodes 20 --stochastic-expert \
  --expert-dir results/examples/expert_data_reexported \
  --output results/examples/sac_reexport
```

The environment, network widths, and architecture options must match the trained
policy. Include `--absorbing` when restoring a vector policy trained with that
flag; the SAC expert above uses ordinary observations without an absorbing indicator.

## 2. Train GAIL, AIRL, EQN-GAIL, and EQN-AIRL

The imitation launcher builds a **new SAC generator** and a reward learner. It
loads demonstrations from SAC; it does not initialize the generator from the SAC
expert checkpoint. The generator learns from discriminator rewards, while test
episodes report the original environment reward.

| Method | Discriminator inputs | Launcher |
| --- | --- | --- |
| GAIL | State and action | `--algorithm gail` |
| AIRL | State/action transitions and stored action log densities | `--algorithm airl` |
| EQN-GAIL | State and action, with an equation discriminator | `--algorithm eqn-gail` |
| EQN-AIRL | Equation reward plus a dense shaping potential | `--algorithm eqn-airl` |
| GAIfO | State and successor state | Python assembly in section 3 |
| EQN-GAIfO | State and successor state, with an equation discriminator | Python assembly in section 3 |

Set shared options once, then run each command separately:

```bash
export EXPERT_DIR="$PWD/results/examples/expert_data"
COMMON=(
  --env-name Pendulum-v1 --gpu -1 --seed 11
  --expert-path-dir "$EXPERT_DIR" --n-expert-trajectories 20
  --max-steps 20000 --episode-max-steps 200 --n-warmup 1000
  --memory-capacity 50000 --batch-size 128 --batch-size-ail 32
  --units 64,64 --units-ail 64,64 --discriminator-noise 0.0
  --test-interval 5000 --test-episodes 5
  --save-model-interval 5000 --save-summary-interval 100
)

python examples/train_imitation.py "${COMMON[@]}" \
  --algorithm gail --logdir results/examples/gail

python examples/train_imitation.py "${COMMON[@]}" \
  --algorithm airl --state-only --logdir results/examples/airl

python examples/train_imitation.py "${COMMON[@]}" \
  --algorithm eqn-gail --v 2,1 --dropout 0,0 --export-equation \
  --logdir results/examples/eqn-gail

python examples/train_imitation.py "${COMMON[@]}" \
  --algorithm eqn-airl --state-only --v 2,1 --dropout 0,0 --export-equation \
  --logdir results/examples/eqn-airl
```

Notice the argument formats: `train_sac.py` takes `--units 64 64`, while the scalar
imitation recipe takes `--units 64,64`. AIRL uses SAC as its generator because it
needs action log densities. `--state-only` makes the AIRL reward depend on the
state alone; omit it for a state-action reward. GAIL and EQN-GAIL also accept
`--generator td3` or `--generator ddpg`.

The loader appends the absorbing indicator and builds DAC transitions for these
imitation runs. Keep expert and learner observation/action spaces compatible;
do not manually add absorbing self-loops to the exported files. Pendulum has
time-limit truncations, so it will not exercise genuine absorbing terminations.

Scalar imitation runs create a timestamped child directory under each `--logdir`:

```text
results/examples/eqn-gail/GENERATED_RUN_DIRECTORY/
├── args.txt                  # Actual settings, including network widths
├── command.txt
├── checkpoint                # TensorFlow checkpoint-state file
├── ckpt-*.index              # Policy checkpoint prefixes have matching .data files
├── irl/                      # Separate discriminator checkpoints
├── equations/                # EQN exports when --export-equation is enabled
└── events.out.tfevents.*
```

`--v 2,1` controls equation operator repetition counts per layer, not dense hidden
widths. Optional sparse fitting adds, for example, `--eqn-l1 0.001 --eqn-stages
1000,4000 --eqn-prune-threshold 0.01`. Stage counts are **discriminator optimizer
updates**, not environment steps. See [equation learning](../docs/equation_learning.md)
for exported JSON/text/NumPy predictors and restoration settings. These staged
sparsity/export flags apply to EQN-GAIL and EQN-AIRL, not the legacy EQN-GAIfO class.

## 3. Train GAIfO and EQN-GAIfO through Python

Neither `gaifo` nor `eqn-gaifo` is a current CLI algorithm choice. The following
assembly uses the existing environment, agent, demonstration loader, and
`ImitationTrainer` factories, and explicitly selects the observation-only reward
learner. Run the block once with `METHOD=gaifo` and again with `METHOD=eqn-gaifo`.

The retained dense GAIfO implementation needs eager function execution in the
current Keras 3 environment: its compiled path can raise “No gradients provided”.
This example enables eager mode for that method. The legacy equation output
regularizer needs a floating-point coefficient, so EQN-GAIfO uses `lmbda=0.0`.
These paths are useful research starting points, not evidence of benchmark quality.

```bash
export EXPERT_DIR="$PWD/results/examples/expert_data"
export METHOD=gaifo                 # Change to eqn-gaifo for the equation version
python - <<'PY'
import os
from contextlib import ExitStack

import tensorflow as tf

from tf2rl.algos.imitation.gaifo import GAIfO
from tf2rl.algos.imitation.gaifo_eqn import GAIfO_EQL
from tf2rl.envs.factory import make_single_environment
from tf2rl.experiments.build import make_imitation_agent, make_imitation_trainer
from tf2rl.experiments.config import imitation_parser, validate_imitation
from tf2rl.utils.tensorflow import configure_tensorflow

method = os.environ["METHOD"]
if method not in ("gaifo", "eqn-gaifo"):
    raise ValueError("METHOD must be gaifo or eqn-gaifo")
args = imitation_parser().parse_args([
    "--algorithm", "gail",  # Reuse scalar settings; select the actual model below.
    "--env-name", "Pendulum-v1", "--gpu", "-1", "--seed", "11",
    "--expert-path-dir", os.environ["EXPERT_DIR"],
    "--n-expert-trajectories", "20", "--max-steps", "20000",
    "--episode-max-steps", "200", "--n-warmup", "1000",
    "--memory-capacity", "50000", "--batch-size", "128",
    "--batch-size-ail", "32", "--units", "64,64", "--units-ail", "64,64",
    "--discriminator-noise", "0.0", "--test-interval", "5000",
    "--test-episodes", "5", "--save-model-interval", "5000",
    "--save-summary-interval", "100", "--logdir", f"results/examples/{method}",
])
validate_imitation(args)
args.algorithm = method             # Record the actual method in args.txt.
args.debug = method == "gaifo"
configure_tensorflow(args.seed, cpu_threads=1)
tf.config.run_functions_eagerly(args.debug)

with ExitStack() as resources:
    env = make_single_environment(args, imitation=True)
    resources.callback(env.close)
    test_env = make_single_environment(args, evaluation=True, imitation=True)
    resources.callback(test_env.close)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    test_env.reset(seed=args.seed + 1)
    absorbing = args.absorbed_state_disable  # Historical store_false CLI field.
    policy = make_imitation_agent(args, env, imitation=True, absorbing=absorbing)
    common = dict(state_shape=env.observation_space.shape,
                  batch_size=args.batch_size_ail, lr=args.lr_ail, gpu=args.gpu)
    if method == "gaifo":
        irl = GAIfO(**common, units=(64, 64))
    else:
        irl = GAIfO_EQL(**common, action_dim=env.action_space.shape[0],
                       num_layers=2, v=(2, 1), drop_out=None,
                       lmbda=0.0, update_delay=3)
    trainer = make_imitation_trainer(args, policy, irl, env, test_env, absorbing)
    resources.pop_all()             # The initialized trainer now owns both envs.
    resources.callback(trainer.close)
    trainer.run()
PY
```

Although GAIfO's discriminator ignores actions, the shared demonstration format
and trainer still require an `act` array. The SAC exports already contain it.
GAIfO does not require expert `logp` values. Checkpoints and events use the same
timestamped layout as the other scalar imitation runs. EQN-GAIfO retains its
historical implementation; it does not provide the modern equation-export or
staged-pruning interface.

## 4. Load checkpoints and watch the policies

For a numerical evaluation of the vector SAC expert:

```bash
python examples/train_sac.py \
  --env-name Pendulum-v1 --gpu -1 --units 64 64 \
  --evaluate --model-dir results/examples/sac_expert --eval-episodes 5 \
  --output results/examples/sac_evaluation
```

For an interactive SAC viewer, the scalar expert recipe can restore the same
64-wide SAC architecture from its managed checkpoint directory:

```bash
python -m tf2rl.experiments.train --runner imitation --algorithm sac \
  --env-name Pendulum-v1 --gpu -1 --units 64,64 \
  --evaluate --model-dir results/examples/sac_expert/checkpoints \
  --show-test-progress --test-episodes 3 --episode-max-steps 200 \
  --logdir results/examples/playback
```

For an imitation checkpoint, set `RUN_DIR` to the **timestamped run itself**,
then select its original algorithm:

```bash
RUN_DIR=results/examples/eqn-gail/GENERATED_RUN_DIRECTORY
python examples/train_imitation.py \
  --algorithm eqn-gail --env-name Pendulum-v1 --gpu -1 --units 64,64 \
  --evaluate --model-dir "$RUN_DIR" \
  --show-test-progress --test-episodes 3 --episode-max-steps 200 \
  --logdir results/examples/playback
```

Use `--algorithm gail`, `airl`, or `eqn-airl` with its corresponding run directory
for the other CLI methods. For either GAIfO run above, use `--algorithm gail`:
playback reconstructs the same SAC generator and absorbing observation wrapper;
it does not construct a GAIL discriminator or evaluate learned rewards. All these
examples assume the default SAC generator and 64-wide policy used above.

An interactive viewer needs a display. To save GIFs instead, replace
`--show-test-progress` with `--save-test-movie`; on a headless MuJoCo machine,
prefix the command with `MUJOCO_GL=egl`. GIFs are written under the new playback
run directory. The module `python -m tf2rl.training.evaluation --run-dir ...` is
the **on-policy PPO/VPG viewer**; use the commands above for these SAC generators.

Policy playback needs only the policy checkpoint. To continue imitation training
with its learned reward, also pass `--irl-model-dir "$RUN_DIR/irl"` together with
`--model-dir "$RUN_DIR"`, matching algorithm/model/sparsity options and expert
data. This restores tracked model/optimizer state, not replay, environment state,
or the original RNG stream. The GAIfO Python assembly would likewise need explicit
policy and discriminator restoration before `trainer.run()`.

## 5. Monitor training with TensorBoard

Run this in another activated terminal while training is running:

```bash
tensorboard --logdir results/examples --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>. Select the runs you want to compare. Useful tags are
`OffPolicy/evaluation_return` for vector SAC, `Common/average_test_return` for
scalar imitation, `SAC/actor_loss`, `SAC/critic_Q1_loss`,
`GAIL/learned_reward_mean`, and each learner's `DiscriminatorLoss`/`Accuracy` tags
where emitted. AIRL and EQN-AIRL report `AIRL/Loss` and `AIRL_EQL/Loss`, respectively.
Dense GAIfO uses the `GAIfO/` prefix and its equation version uses
`GAIfO_EQL/`. Sparse EQN runs also emit `EQN/` metrics.

For the four CLI imitation runs, inspect saved records without retraining:

```bash
python scripts/inspect_training_records.py results/examples/gail
python scripts/inspect_training_records.py results/examples/eqn-gail
```

For a reproducible small SAC → GAIL → EQN-GAIL integration run, see
[validation](../docs/validation.md) and `scripts/run_validation_pipeline.py`.
