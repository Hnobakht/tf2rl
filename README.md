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

**Validation status:** SAC demonstration generation, dense GAIL and EQN-GAIL have
working training, evaluation, checkpoint and TensorBoard paths. Unit-test suites
were removed in the 2026-10-16 cleanup; executable experiment smoke workflows remain.
See [executed experiments and limits](docs/validation.md) and the
[cleanup record](docs/cleanup.md). Broader experimental algorithms remain unverified.

## Installation

Use Python 3.10–3.12 and a virtual environment inside the checkout:

```bash
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export KERAS_HOME="$PWD/.cache/keras"
python -m venv .venv-modern
source .venv-modern/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the editable project with MuJoCo, rendering and
development extras, pinning direct dependencies and key runtime packages to the
versions validated in `.venv-modern` on Python 3.10/Linux. It is not a complete
transitive dependency lock. For a smaller installation using compatible version
ranges, run `python -m pip install -e '.[mujoco]'`; add `render` for pixel
observations, classic-control rendering or GIF export.
No separately modified Gym installation is required. `legacy/gym`
is a provenance archive and must not be installed for these experiments.

Dependencies are declared in `pyproject.toml`: TensorFlow 2.18, TensorFlow Probability
0.25, tf-keras 2.18, Gymnasium 1.2–1.3, NumPy 1.26–2.0, cpprb 11, SciPy and joblib.
TFP 0.25 is tested upstream against TensorFlow 2.18 and requires the separate
`tf-keras` package. Our model code uses `tf.keras`; no global legacy-Keras switch
is set. [Upstream compatibility notes](https://github.com/tensorflow/probability/releases/tag/v0.25.0).

TensorFlow Addons, `tensorflow-gpu`, Python 2 compatibility helpers and `mujoco-py`
are unnecessary for the modern research path. Spectral normalization is implemented
locally using the historical power-iteration rule. Native MuJoCo is optional.
The TensorBoard 2.18 CLI needs `pkg_resources`, so the runtime declares
`setuptools>=77,<82`; upgrading setuptools beyond this range breaks the dashboard.

## Validated three-stage workflow and TensorBoard

The following command runs 20,000 steps each of SAC, GAIL and EQN-GAIL on
`Pendulum-v1`, using 200-step episodes. The final 20 SAC evaluation trajectories
are shared by both imitation runs. Each stage saves four checkpoints and periodic
TensorBoard metrics. It also checks event values and checkpoint contents:

```bash
.venv-modern/bin/python scripts/run_validation_pipeline.py
.venv-modern/bin/tensorboard --logdir results/validated_pipeline --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>. Select runs under `sac/`, `gail/`, and `eqn-gail/`.
Useful scalar tags include `Common/average_test_return`, `SAC/actor_loss`,
`SAC/critic_Q1_loss`, `GAIL/DiscriminatorLoss`, `GAIL_EQL/DiscriminatorLoss`,
`GAIL/learned_reward_mean`, and `DAC/absorbing_batch_fraction`.
TensorBoard displays TensorFlow 2 scalar events even though its event-reader API
classifies them as tensor summaries.

Pendulum episodes truncate at a time limit; zero absorbing/terminal fractions are
expected. Before removal, integration checks exercised genuine termination,
including termination coincident with truncation. These runs check one task and one seed per
algorithm; they do not reproduce the paper's full benchmark results.

To check existing records without retraining:

```bash
.venv-modern/bin/python scripts/inspect_training_records.py results/validated_pipeline
```

Run directories retain `command.txt`, `args.txt`, checkpoints, event files, and
source revision/diff metadata. `validation_summary.json` contains all metric
counts, last steps, finite-value checks and evaluation curves.

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

## Debugging EQN operators in VS Code

Open the repository folder in VS Code with the Python Debugger extension installed.
Choose **EQN-GAIL: HalfCheetah (eager debug)** from Run and Debug, then press F5.
The configuration in `.vscode/launch.json` uses `.venv-modern`, the saved HalfCheetah
expert trajectories, and a fresh 2,000-step training run with 256 warmup steps.
Adjust `--expert-path-dir` if your demonstrations are in a different directory.
TensorBoard records go under `results/debug/eqn-gail-halfcheetah`.

In `tf2rl/networks/eqn.py`, set breakpoints on the `value = tf.math.log(...)` and
`value = (a + 0.1) / (...)` lines inside `EquationLayer.call`. Right-click each
breakpoint, choose Edit Breakpoint, and use this expression to skip initialization
and stop on eager discriminator training calls:

```python
training is True and tf.executing_eagerly()
```

Inspect `a.numpy()` (and `b.numpy()` for division) in the Debug Console. Press F10
to execute the operation, then inspect `value.numpy()`. `self.name` identifies the
equation layer and `group` identifies the operator repetition. F5 continues to the
next breakpoint; disable breakpoints when you want training to progress freely.

The CLI's `--debug` enables TensorFlow eager function execution so Python
breakpoints run on each batch instead of only during graph tracing. This is slower
and cannot be combined with `--xla`; the equations and loss functions are unchanged.
`--is-debug` separately enables discriminator weight histograms in TensorBoard.

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

## Project structure and validation

- `tf2rl/algos`: original algorithms, shared research losses, EQN-AIRL/EQN-GAIL and SAC.
- `tf2rl/networks/eqn.py`: equation operators, scalar output, and reusable networks.
- `tf2rl/envs`: native wrappers, custom tasks and packaged `assets/`.
- `tf2rl/experiments`: common CLI, DAC trainer, demonstration I/O and expert sampling.
- `scripts`: executable training workflows and TensorBoard/checkpoint inspection.
- `legacy/python`: distinct historical algorithms and launchers, excluded from the
  runtime package. The modern trainers live in `irl_trainer.py` and `airl_trainer.py`;
  their old `_v2` import paths are compatibility aliases.
- `legacy/gym`: historical Gym source and assets, excluded from the runtime package.

```bash
python -m compileall -q tf2rl examples scripts
bash scripts/smoke_examples.sh
python scripts/run_validation_pipeline.py --steps 300 --warmup 32 --interval 300 --units 32,32 --output results/smoke-pipeline
python -m build
```

Unit tests and the pytest dependency have been removed. Historical validation
results and limitations remain in [docs/validation.md](docs/validation.md).
Original upstream documentation is retained in
[docs/upstream-readme.md](docs/upstream-readme.md). Results from the paper
cannot be reproduced or claimed verified without its demonstrations, experiment
configurations, original physics/framework versions and full-length training runs.
