# Running research experiments

The canonical assembly is `python -m tf2rl.experiments.train`. Choose one of three
recipes with `--runner off_policy|on_policy|imitation`. Each recipe keeps its existing
arguments and defaults. Training loops live in `tf2rl/training/`.

```bash
.venv-modern/bin/python -m tf2rl.experiments.train --runner off_policy \
  --algorithm sac --env-name Pendulum-v1 --max-steps 100000

.venv-modern/bin/python -m tf2rl.experiments.train --runner on_policy \
  --algorithm ppo --env-name Pendulum-v1 --iterations 20

.venv-modern/bin/python -m tf2rl.experiments.train --runner imitation \
  --algorithm eqn-gail --env-name Pendulum-v1 --expert-path-dir PATH_TO_EXPERT
```

`eqn-train`, `offpolicy-train`, `onpolicy-train`, and `ppo-train` invoke the shared
assembly directly. Examples are `train_sac.py`, `train_ppo.py`, `train_imitation.py`,
and `custom_algorithm.py`. Run each experiment in a fresh process when changing
TensorFlow device or thread settings.

## Hydra and sweeps

`tf2rl-train` composes a preset and launches the canonical assembly in a child
process. Presets live directly under `tf2rl/experiments/conf/experiment/`; each
contains its recipe and full editable argument defaults.

```bash
tf2rl-train experiment=sac args.env_name=Pendulum-v1 args.max_steps=100000
tf2rl-train -m experiment=ppo args.env_name=Pendulum-v1 args.seed=0,1,2
tf2rl-train experiment=eqn_gail args.env_name=Pendulum-v1 args.expert_path_dir=PATH
```

Presets: `sac`, `td3`, `ddpg`, `ppo`, `vpg`, `expert_sac`, `airl`, `gail`,
`eqn_airl`, `eqn_gail`. Algorithm values such as `--algorithm eqn-gail` retain
their hyphenated spellings. Old preset aliases and `conf/runner` templates have
been removed; custom configurations can extend an experiment preset.

`expert_sac` preserves the original scalar expert recipe (128-wide default
networks); `sac` selects the vector recipe (64-wide default networks). YAML
booleans express CLI flag presence, including historical `*_disable` flags.
`null` retains a dynamic CLI default. Vector widths are lists; imitation widths
and equation schedules retain comma-separated strings.

Hydra records resolved configuration and the child command. Relative input paths
remain relative to the invocation directory. Each run retains its existing
artifact layout: scalar runs use timestamped subdirectories; vector runs write
checkpoints, events, config and metrics directly under their output directory.

## Expert generation and evaluation

Use the scalar expert recipe with `--save-test-path`, or vector off-policy training
with `--save-expert`. Demonstrations store chronological physical transitions,
termination/truncation metadata, and action log probabilities when available.
AIRL requires stored log probabilities. Loader conversion preserves legacy files
and DAC augmentation without duplicating absorbing indicators.

The off-policy and imitation recipes support `--evaluate --model-dir PATH`.
Supply matching network and algorithm options. Model restore includes tracked
optimizer and target-network state; replay, environment and RNG streams are not
resumed. Equation models also require matching sparsity options.

Use `--env-backend mjwarp` for supported GPU tasks. `--gpu` selects the learner;
`--env-gpu` selects simulation. [MJWarp details](mjwarp.md) and the existing
on/off-policy backend guides describe task limits and dependencies.

See [architecture](architecture.md), [adding algorithms](adding_algorithms.md),
[equation learning](equation_learning.md), and [validation](validation.md).
The earlier configuration/refactor record is retained in
[history](history/experiments-before-architecture.md).
