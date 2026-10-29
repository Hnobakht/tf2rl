# Configuring and extending experiments

Use `tf2rl-train` (or `python -m tf2rl.experiments.hydra`) to compose experiments
with Hydra. Existing `eqn-train`, `onpolicy-train`, `offpolicy-train`, `ppo-train`
and example commands keep their arguments and defaults.

After updating a checkout, install the new dependency and console command:

```bash
python -m pip install -e .
tf2rl-train --help
tf2rl-train --cfg job experiment=ppo args.env_name=Pendulum-v1
```

The environment ID is required. Configuration inspection does not initialize
TensorFlow. Presets select the existing launcher and algorithm:

| Preset (`experiment=...`) | Existing execution path |
| --- | --- |
| `sac`, `td3`, `ddpg` | Vector off-policy launcher |
| `ppo`, `vpg` | Vector on-policy launcher |
| `expert-sac` | Original single-environment SAC expert launcher |
| `airl`, `gail`, `eqn-airl`, `eqn-gail` | Original imitation launcher |

`expert-sac` retains the expert launcher's settings, including its 128-wide
networks; `sac` retains the vector launcher's settings, including its 64-wide
networks. Changing presets is an experiment choice, not a migration of a saved run.
Historical algorithms and specialized examples outside these launchers remain
available through their original Python interfaces.

## Runs and sweeps

```bash
tf2rl-train experiment=sac args.env_name=Pendulum-v1 args.max_steps=100000

tf2rl-train -m experiment=ppo args.env_name=Pendulum-v1 \
  args.seed=0,1,2 args.learning_rate=0.0003,0.001

tf2rl-train -m experiment=sac,td3 args.env_name=Pendulum-v1 \
  args.seed=0,1 args.max_steps=100000
```

Hydra launches a fresh Python process for each experiment using the same
interpreter. The default sweeper runs jobs sequentially, releasing TensorFlow GPU
memory and device/thread configuration between runs. A failing child fails its
Hydra job. There is no second training loop or scheduler.

Single runs go to `results/hydra/ALGORITHM/DATE/TIME`; sweep jobs go to numbered
subdirectories of `results/hydra/multirun/DATE/TIME`. Set `hydra.run.dir=PATH` or
`hydra.sweep.dir=PATH` to choose another location. Each job contains:

- `.hydra/`: Hydra configuration, launcher settings, and overrides.
- `resolved.yaml`: the composed configuration with interpolations resolved.
- `command.json`: the exact child command as an argument list.
- `artifacts/`: existing checkpoints, TensorBoard events, and training records.

The imitation trainer retains its timestamped subdirectory under `artifacts/`.
Vector launchers write directly to `artifacts/`. Their `config.json`, or the
imitation trainer's `args.txt`, records effective arguments after CLI defaults
and algorithm-specific defaults have been applied. Child progress is printed to
the terminal; Hydra's own log is not a copy of the child's stdout.

Working-directory changes are disabled. Relative dataset and checkpoint paths
are always interpreted from the directory where the command was invoked, even
if `hydra.job.chdir=true` is explicitly selected. Output locations are absolute
when passed to the training process. Keep the default per-job artifact paths for
sweeps so jobs do not overwrite one another's records.

## Configuration values

`args` mirrors CLI option names with underscores in place of hyphens. The YAML
files in `tf2rl/experiments/conf/runner` expose the existing defaults for each
launcher; the small files in `conf/experiment` select an algorithm. The CLI still
owns argument conversion and validation. Keep the YAML defaults aligned when
deliberately changing a launcher default.

- `null` leaves an option at its CLI default. This preserves dynamic defaults,
  such as PPO's ten epochs versus VPG's one epoch and EQN-AIRL's potential widths.
- Booleans indicate **flag presence**. `args.absorbed_state_disable=true` disables
  absorbing transitions; `args.auto_alpha_disable=true` disables adaptive alpha.
  These historical names have not been silently inverted. Vector SAC instead
  uses `args.absorbing=true` and `args.fixed_alpha=true`.
- Vector hidden widths use YAML lists: `'args.units=[64,64]'`.
- Imitation widths, EQN operators and schedules retain their comma-separated CLI
  strings: `'args.units="128,128"'`, `'args.v="2,1"'`,
  `'args.eqn_stages="1000,4000"'`. The inner quotes distinguish a string from a
  Hydra sweep.
- Environment and policy keyword arguments use mappings, e.g.
  `+args.env_kwargs.gravity=9.81` or `+args.policy_kwargs.entropy_coef=0.01`.
  A leading `+` adds a new mapping key; ordinary overrides reject misspelled keys.

When sweeping over a configuration group, Hydra first validates overrides against
the application's default preset (`sac`). For example, a `experiment=ppo,vpg`
sweep needs `++args.rollout_steps=1024` to add or override an on-policy-only key.
For a seed or parameter sweep within PPO, use `experiment=ppo` and ordinary
`args.rollout_steps=1024`. See Hydra's
[multirun documentation](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/)
and [output-directory configuration](https://hydra.cc/docs/1.3/configure_hydra/workdir/).

To maintain a project-specific preset outside the package, create
`configs/experiment/my_ppo.yaml`:

```yaml
# @package _global_
defaults:
  - /runner: on_policy
  - _self_
args:
  algorithm: ppo
  env_name: HalfCheetah-v5
  num_envs: 8
  rollout_steps: 256
  batch_size: 256
```

Then run `tf2rl-train --config-dir configs experiment=my_ppo args.seed=7`.
The package defaults supply the remaining options. Archive custom presets with
the experiment's source revision.

## Demonstrations and continued training

```bash
tf2rl-train experiment=expert-sac args.env_name=Pendulum-v1 \
  args.max_steps=100000 args.test_interval=50000 args.test_episodes=20 \
  args.save_test_path=true hydra.run.dir=results/expert-hydra

tf2rl-train experiment=eqn-airl args.env_name=Pendulum-v1 args.state_only=true \
  args.expert_path_dir=results/expert-hydra/artifacts/GENERATED_RUN_DIRECTORY

tf2rl-train experiment=sac args.env_name=Pendulum-v1 args.evaluate=true \
  args.model_dir=results/PREVIOUS_RUN/artifacts
```

The same checkpoint architecture and restore restrictions apply as with the CLI.
Restoring model weights does not resume replay, environment state, RNG streams,
or Hydra's sweep position. AIRL still requires demonstrations with log probabilities.

## Existing TensorFlow GPU container

Use the Python environment that already contains the project's dependencies.
Do not replace the container's TensorFlow/CUDA stack just to enable Hydra. For
the existing container and its isolated MJWarp environment:

```bash
docker exec -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -m pip install 'hydra-core>=1.3,<1.4'

docker exec -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.hydra \
  experiment=ppo args.env_name=HalfCheetah-v5 args.env_backend=mjwarp \
  args.gpu=0 args.env_gpu=0 args.num_envs=64 args.rollout_steps=32
```

`args.gpu` selects the learner device; `args.env_gpu` selects MJWarp simulation.
Use `args.gpu=-1` for CPU learning with GPU simulation. Gymnasium remains the
default simulation backend. Container paths are interpreted inside the container.

## Code organization and extension

The launchers expose `build_parser(algorithm=None)`, `run(args)`, and the compatible
`main(argv=None, algorithm=None)`. `run` receives a parsed `argparse.Namespace`,
applies existing validation and dynamic defaults, and owns execution and cleanup.
It raises `ConfigurationError` for invalid experiment settings; `main` converts
that into the existing CLI error behavior. Run each experiment in a fresh process
when changing TensorFlow devices or thread settings.

`imitation.py` contains ordinary factory functions for the SAC generator,
discriminator, demonstration/trainer assembly, and discriminator restoration.
`runtime.py` shares vector-launcher TensorFlow setup and argument recording.
`hydra.py` only composes, records, and launches experiments. These small boundaries
keep configuration and construction out of algorithm code without adding a class
hierarchy or dependency-injection framework.

The existing on-policy registration API and off-policy factories remain available.
Custom in-process registrations belong in a Python entry point that imports them
before calling the launcher; they are not inherited by Hydra's child processes.
When extending the packaged Hydra entry point, make the registration available in
the selected child launcher and add a preset. Model losses, update order, replay
masks, collection boundaries, and evaluation rules remain owned by the existing
algorithms and trainers.

## Refactor validation (2026-10-23)

Validation used the existing CPU environment (TensorFlow 2.18.1) and the running
GPU container's isolated environment (TensorFlow 2.21.0). Hydra 1.3.7 was installed
in both; the container's TensorFlow/CUDA versions were preserved.

- All ten presets parsed to the same defaults as their existing CLI equivalents.
  The three extracted argument builders were structurally identical to their
  original parser statements.
- Matched before/after CPU runs produced exactly equal values for 810 checkpoint
  tensors, including optimizer and target-network state. Coverage included vector
  SAC (32 steps), PPO (two 16-transition iterations), and 16-step runs of expert
  SAC, AIRL, GAIL, EQN-AIRL and EQN-GAIL, including discriminator checkpoints.
  Matching PPO and vector SAC runs through Hydra also reproduced all 183 original
  CLI checkpoint tensors exactly.
- The existing 300-step-per-stage SAC → GAIL → EQN-GAIL pipeline completed and
  its record inspector reported finite metrics. Existing example smoke commands
  passed. The existing unittest suite ran 23 checks: 12 passed and 11 requiring
  the optional GPU runtime were skipped locally. No tests or test dependencies
  were added or modified.
- Hydra completed CPU sweeps of SAC/TD3 and PPO/VPG, two seeds each; two GPU SAC
  jobs; and a GPU PPO/MJWarp HalfCheetah run with 32 transitions and four minibatch
  updates. Device records confirmed GPU actors/critics and CUDA simulation.
- Sparse EQN-GAIL training/export completed through Hydra with comma-separated
  settings, a relative demonstration path, an output path containing a space,
  and `hydra.job.chdir=true`.
- Compilation, lint of changed Python modules, CLI help, custom preset composition,
  and wheel/sdist builds passed. The wheel contains all 14 YAML configurations and
  ran a Hydra training job from outside the checkout.

Local records are under `results/hydra-refactor/`; diagnostic logs and the
before/after checkpoint comparisons are under `.work/hydra-refactor/`. These short
runs establish refactor compatibility for the exercised paths, not learning
performance, long-run convergence, or equivalence across TensorFlow versions.
