# EQN-AIRL and EQN-GAIL

Research implementations of Equation Network adversarial imitation learning with
embodiment mismatch, built on the original TensorFlow 2 TF2RL implementation.
The historical Python names `AIRL_EQL` and `GAIL_EQL` remain available; the research
method is called EQN in the paper.

The modernization preserves the implemented AIRL/GAIL objectives, EQN operators,
agent-only gradient penalty, delayed discriminator updates, expert/agent mixing,
and DAC absorbing-state targets. It does **not** substitute a standard-library
imitation algorithm. [Scientific details and behavioral repairs](docs/modernization.md)
explain differences between the executable baseline, the attached paper, and API fixes.

**Validation status:** environment/data tests passed locally. TensorFlow, replay-buffer
and native MuJoCo execution could not be validated in this session because the required
packages were unavailable and network downloads failed. The model/training tests are
included and run by the research CI workflow. See [validation](docs/validation.md).

## Installation

Use Python 3.10–3.12 and a virtual environment inside the checkout:

```bash
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python -m venv .venv-modern
source .venv-modern/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,mujoco]'
```

Install `.[render]` additionally for pixel observations, classic-control rendering,
or GIF export. No separately modified Gym installation is required. `legacy/gym`
is a provenance archive and must not be installed for these experiments.

Dependencies are declared in `pyproject.toml`: TensorFlow 2.18, TensorFlow Probability
0.25, tf-keras 2.18, Gymnasium 1.2–1.3, NumPy 1.26–2.0, cpprb 11, SciPy and joblib.
TFP 0.25 is tested upstream against TensorFlow 2.18 and requires the separate
`tf-keras` package. Our model code uses `tf.keras`; no global legacy-Keras switch
is set. [Upstream compatibility notes](https://github.com/tensorflow/probability/releases/tag/v0.25.0).

TensorFlow Addons, `tensorflow-gpu`, Python 2 compatibility helpers and `mujoco-py`
are unnecessary for the modern research path. Spectral normalization is implemented
locally using the historical power-iteration rule. Native MuJoCo is optional.

## Experiments

The common entry point is `eqn-train`, or `python -m tf2rl.experiments.cli`.
Run `eqn-train --help` to see options. The existing four AIRL/GAIL example filenames
forward to the same implementation and keep their corresponding algorithm selection.
Training is stochastic; evaluation uses the policy mean. `--seed` sets Python, NumPy,
TensorFlow and environment seeds. Different framework/physics versions may still
produce different learning curves.

Generate ordered demonstrations with the original expert SAC architecture:

```bash
eqn-train --algorithm sac --env-name Pendulum-v1 \
  --max-steps 100000 --test-interval 50000 --test-episodes 20 \
  --save-test-path --logdir results/expert
```

Choose the generated directory containing `step_*_epi_*_return_*.pkl` files:

```bash
eqn-train --algorithm eqn-airl --env-name Pendulum-v1 \
  --expert-path-dir results/expert/GENERATED_RUN_DIRECTORY --state-only \
  --v 2,1 --dropout .5,0 --update-delay 3 --logdir results/eqn-airl

eqn-train --algorithm eqn-gail --env-name Pendulum-v1 \
  --expert-path-dir results/expert/GENERATED_RUN_DIRECTORY \
  --v 2,1 --dropout .5,0 --update-delay 3 --logdir results/eqn-gail
```

Use `--algorithm airl` or `--algorithm gail` for the dense baselines. The default
EQN-GAIL gradient-penalty coefficient is zero, matching the historical constructor;
`--gp-coef` explicitly enables it. `--discriminator-noise` defaults to 0.15.
`--random-epoch-n` enables periodic warmup-replay expert substitution.
`--absorbed-state-disable` disables absorbing transitions and the corresponding
SAC target. The indicator remains part of imitation observations in this mode.

`--exclude` specifies exclusions per EQN layer, separated by commas; operators in a
layer are separated by `-`, e.g. `--exclude 'log-div,id-sin-cos-sig-log-mult'`.
Layer counts must match `--v` and the dropout list. The historical width default is
`2,1`; AIRL's dense potential defaults to `64,64` in the EQN CLI.

Custom environment IDs remain registered on `import tf2rl`, including
`pointmaze-left-v0`, `pointmaze-right-v0`, `hopper-longleg-v0`,
`halfcheetah-longbody-v0`, `ant-disabled-v0`, and `walker2d-disabled-v0`.
Their original XML assets, observation ordering, and rewards are retained. Install
`.[mujoco]` to use them. Standard environments use Gymnasium IDs such as `Pendulum-v1`
and `HalfCheetah-v5`; environment version changes need independent reproduction checks.

Evaluate policy checkpoints with the same algorithm/environment and architecture:

```bash
eqn-train --algorithm eqn-airl --env-name Pendulum-v1 --evaluate \
  --model-dir results/eqn-airl/GENERATED_RUN_DIRECTORY --test-episodes 5
```

Policy checkpoints are stored in the run directory; discriminator checkpoints are
stored under `irl/`. `--model-dir` and `--irl-model-dir` load them for continued
training. Restoring weights/optimizers does not restore an in-progress environment,
replay contents or RNG stream. Older TensorFlow checkpoints may need explicit weight
conversion; do not assume compatibility across the major Keras change.

For learned reward evaluation, construct `LearnedRewardWrapper(env, irl)` using a
restored discriminator. Its native `step(action)` retains the environment reward in
`info['environment_reward']`. Historical external-reward wrapper classes and their
sign conventions remain as experimental source; `--reward-dir` is not silently ignored.

## Absorbing states and data

`AbsorbingWrapper` appends a zero indicator to normal vector observations. The
absorbing observation is `[0, ..., indicator]`, with experiment default indicator 0.1.
The environment itself returns ordinary Gymnasium reset/step results.

The collector and expert loader share `dac_transitions`:

| Transition | Replay mask (`done`) | Next state |
| --- | ---: | --- |
| Ordinary / time-limit truncation | 1 | Observed next state |
| True termination | 0 | Absorbing state when enabled |
| Absorbing self-loop | -1 | Same absorbing state; zero action |

These masks are **not boolean done flags**. The absorbing SAC target bootstraps
terminal/self-loop transitions at zero action, suppresses their next-state entropy,
and excludes self-loops from actor updates. One self-loop is stored per true terminal
transition. If both Gymnasium flags are true, termination wins. Gymnasium's distinction
between termination and truncation supports correct bootstrapping.
[Migration guide](https://gymnasium.farama.org/introduction/migration_guide/).

New exports include `terminated`, `truncated` and `format_version=2`, and contain only
actual transitions in chronological order. Legacy joblib trajectories remain readable.
Where old files lack boundary metadata, a done at the environment horizon is inferred
as truncation with a warning. Already augmented observations are not augmented twice.
AIRL requires stored `logp` values; the loader does not invent missing probabilities.

## Project structure and tests

- `tf2rl/algos`: original algorithms, shared research losses, EQN-AIRL/EQN-GAIL and SAC.
- `tf2rl/networks/eqn.py`: equation operators, scalar output, and reusable networks.
- `tf2rl/envs`: native wrappers, custom tasks and packaged `assets/`.
- `tf2rl/experiments`: common CLI, DAC trainer, demonstration I/O and expert sampling.
- `tests/research`: analytic losses/gradients, model updates, transition semantics,
  export/import/training/checkpoint workflows and optional MuJoCo tasks.
- `examples/old`, distinct historical algorithm variants, and `legacy/gym`: preserved
  research/upstream history; not all experimental paths are certified on the new stack.

```bash
python -m pytest tests/research tests/misc tests/envs/test_utils.py --basetemp=.tmp/pytest -q
python -m pytest tests --basetemp=.tmp/pytest-all -q
bash test_run_examples.sh
python -m build
```

The full upstream suite includes optional Atari/DMC and older algorithms; interpret
failures separately from the core research suite. Original upstream documentation is
retained in [docs/upstream-readme.md](docs/upstream-readme.md). Results from the paper
cannot be reproduced or claimed verified without its demonstrations, experiment
configurations, original physics/framework versions and full-length training runs.
